# ... (previous code remains the same)

@app.route('/get_personality_summary', methods=['GET'])
def get_personality_summary():
    telegram_user_id = request.args.get('telegram_user_id')
    if not telegram_user_id:
        return jsonify({'error': 'Missing telegram_user_id parameter'}), 400

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT personality_type, explanation, compatible_types, dating_advice
                    FROM personality_summaries
                    WHERE telegram_user_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (telegram_user_id,))
                result = cur.fetchone()

        if result:
            personality_type, explanation, compatible_types, dating_advice = result
            return jsonify({
                'personality_type': personality_type,
                'explanation': explanation,
                'compatible_types': compatible_types,
                'dating_advice': dating_advice
            })
        else:
            return jsonify({'error': 'No personality summary found for this user'}), 404
    except Exception as e:
        logger.error(f"Error retrieving personality summary: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get_match_suggestions', methods=['GET'])
def get_match_suggestions():
    telegram_user_id = request.args.get('telegram_user_id')
    if not telegram_user_id:
        return jsonify({'error': 'Missing telegram_user_id parameter'}), 400

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get the user's personality type
                cur.execute("""
                    SELECT personality_type, compatible_types
                    FROM personality_summaries
                    WHERE telegram_user_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (telegram_user_id,))
                user_result = cur.fetchone()

                if not user_result:
                    return jsonify({'error': 'No personality summary found for this user'}), 404

                user_personality_type, compatible_types = user_result

                # Find potential matches based on compatible types
                cur.execute("""
                    SELECT ps.telegram_user_id, u.first_name, u.photo_url, ps.personality_type
                    FROM personality_summaries ps
                    JOIN users u ON ps.telegram_user_id = u.telegram_user_id
                    WHERE ps.personality_type = ANY(%s)
                    AND ps.telegram_user_id != %s
                    ORDER BY ps.created_at DESC
                    LIMIT 5
                """, (compatible_types, telegram_user_id))
                matches = cur.fetchall()

        if matches:
            return jsonify([
                {
                    'telegram_user_id': match[0],
                    'first_name': match[1],
                    'photo_url': match[2],
                    'personality_type': match[3]
                } for match in matches
            ])
        else:
            return jsonify({'message': 'No potential matches found'}), 404
    except Exception as e:
        logger.error(f"Error retrieving match suggestions: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Capybara Dating App API")
    app.run(debug=True)